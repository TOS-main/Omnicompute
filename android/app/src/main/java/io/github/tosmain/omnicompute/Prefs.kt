package io.github.tosmain.omnicompute

import android.content.Context

/** The app's few settings. Hosting is the point of the app; computing is an opt-in side job. */
class Prefs(context: Context) {
    private val p = context.getSharedPreferences("omnicompute", Context.MODE_PRIVATE)

    var compute: Boolean
        get() = p.getBoolean("compute", false)
        set(v) = p.edit().putBoolean("compute", v).apply()

    var cores: Int  // 0 = all; the web page's core picker sets it for a running host
        get() = p.getInt("cores", 0)
        set(v) = p.edit().putInt("cores", v).apply()

    var startAtBoot: Boolean
        get() = p.getBoolean("boot", true)
        set(v) = p.edit().putBoolean("boot", v).apply()

    var ownIp: Boolean  // host on this phone's own address instead of a Cloudflare tunnel
        get() = p.getBoolean("own_ip", false)
        set(v) = p.edit().putBoolean("own_ip", v).apply()

    var ownUrl: String  // with ownIp: an address from other networks (port forwarding), or "" for Wi-Fi only
        get() = p.getString("own_url", "") ?: ""
        set(v) = p.edit().putString("own_url", v).apply()

    var mode: String  // "" = not chosen yet, "host" = this device is the main one, "help" = it helps another's mesh
        get() = p.getString("mode", "") ?: ""
        set(v) = p.edit().putString("mode", v).apply()

    var helpLink: String  // with mode "help": the contribute link (or permanent link) of the mesh to help
        get() = p.getString("help_link", "") ?: ""
        set(v) = p.edit().putString("help_link", v).apply()

    val helping: Boolean get() = mode == "help" && helpLink.isNotEmpty()

    var hosting: Boolean  // false after the user pressed Stop, so a reboot doesn't start it again
        get() = p.getBoolean("hosting", true)
        set(v) = p.edit().putBoolean("hosting", v).apply()
}
