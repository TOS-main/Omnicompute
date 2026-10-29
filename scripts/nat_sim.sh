#!/usr/bin/env bash
# Simulates devices on different home networks / carriers with Linux network namespaces (needs root + iproute2):
#
#   wan (the "internet", 10.0.0.0/24): hub at 10.0.0.1
#   home1: cone NAT (10.0.0.2)      -> laptop  192.168.1.2
#   home2: cone NAT (10.0.0.3)      -> desktop 192.168.2.2
#   cell:  symmetric NAT (10.0.0.4, random port per destination, like many CGNATs) -> phone 100.64.0.2
#
# Expected: laptop <-> desktop hole-punch to a direct UDP path; anything involving the
# symmetric-NAT "phone" falls back to the hub relay; the generation still completes.
set -euo pipefail
PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS=${LOGS:-$(mktemp -d)}
KEY=simkey-0001

cleanup() {
  for p in $(jobs -p); do kill "$p" 2>/dev/null || true; done
  for ns in wan home1 home2 cell laptop desktop phone; do ip netns del $ns 2>/dev/null || true; done
}
trap cleanup EXIT
cleanup

for ns in wan home1 home2 cell laptop desktop phone; do ip netns add $ns; ip -n $ns link set lo up; done
ip -n wan link add br0 type bridge; ip -n wan link set br0 up; ip -n wan addr add 10.0.0.1/24 dev br0

router() { # name wan_ip lan_dev_ns lan_net symmetric
  local r=$1 wip=$2 dev=$3 net=$4 sym=$5
  ip link add "w-$r" netns wan type veth peer name wan0 netns "$r"
  ip -n wan link set "w-$r" master br0 up
  ip -n "$r" addr add "$wip/24" dev wan0; ip -n "$r" link set wan0 up
  ip link add lan0 netns "$r" type veth peer name eth0 netns "$dev"
  ip -n "$r" addr add "$net.1/24" dev lan0; ip -n "$r" link set lan0 up
  ip -n "$dev" addr add "$net.2/24" dev eth0; ip -n "$dev" link set eth0 up
  ip -n "$dev" route add default via "$net.1"
  ip netns exec "$r" sysctl -qw net.ipv4.ip_forward=1
  # like a real home router's firewall: drop unsolicited traffic to the router itself
  ip netns exec "$r" iptables -A INPUT -i wan0 -m conntrack --ctstate NEW -j DROP
  if [ "$sym" = 1 ]; then
    ip netns exec "$r" iptables -t nat -A POSTROUTING -o wan0 -j MASQUERADE --random-fully
  else
    ip netns exec "$r" iptables -t nat -A POSTROUTING -o wan0 -j MASQUERADE
  fi
}
router home1 10.0.0.2 laptop 192.168.1 0
router home2 10.0.0.3 desktop 192.168.2 0
router cell 10.0.0.4 phone 100.64.0 1

export PYTHONPATH="$ROOT" OMNICOMPUTE_HOME="$LOGS/home" OMNICOMPUTE_BEACON=off
ip netns exec wan $PY -m omnicompute -v host --key $KEY --no-worker --port 8000 >"$LOGS/hub.log" 2>&1 &
sleep 2
declare -A SCORE=([laptop]=2 [desktop]=1 [phone]=3)  # chain laptop -> desktop -> phone (the fastest goes last)
for d in laptop desktop phone; do
  OMNICOMPUTE_SCORE=${SCORE[$d]} ip netns exec $d $PY -m omnicompute -v join "http://10.0.0.1:8000/?k=$KEY" --name $d --always --cpu 100 >"$LOGS/$d.log" 2>&1 &
done
sleep 3
ip netns exec wan $PY -m omnicompute run "http://10.0.0.1:8000/?k=$KEY" "hello from three networks" --max-new 20 \
  >"$LOGS/run.out" 2>"$LOGS/run.err"
cat "$LOGS/run.out"; echo; cat "$LOGS/run.err"
ip netns exec wan $PY -m omnicompute status "http://10.0.0.1:8000/?k=$KEY" | grep -E '"node_id"|public_udp' -A2 | grep -vE 'kind|role|platform|score|mem' || true
sleep 4  # let the background punch attempts to the symmetric NAT time out
echo "--- path decisions ---"
grep -hE "direct UDP path|hole punching failed|falling back" "$LOGS"/laptop.log "$LOGS"/desktop.log "$LOGS"/phone.log || true
grep -q "tok/s" "$LOGS/run.err"
