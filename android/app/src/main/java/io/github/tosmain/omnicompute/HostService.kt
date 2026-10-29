package io.github.tosmain.omnicompute

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File

/**
 * Runs the mesh's main device (omnicompute/android.py) while the screen is off. Android shows its
 * notification the whole time; Stop there (or in the app) ends it.
 */
class HostService : Service() {
    private var wake: PowerManager.WakeLock? = null
    private var wifi: WifiManager.WifiLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            Prefs(this).hosting = false
            stopHosting()
            return START_NOT_STICKY
        }
        Prefs(this).hosting = true
        val n = notification()
        if (Build.VERSION.SDK_INT >= 34) startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(1, n)
        // the hub mostly waits for messages; these keep the phone answering with the screen off
        if (wake == null) {
            wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "omnicompute:host").apply { setReferenceCounted(false); acquire() }
            @Suppress("DEPRECATION")
            wifi = (applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager)
                .createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "omnicompute:host").apply { setReferenceCounted(false); acquire() }
        }
        val prefs = Prefs(this)
        Thread {
            val m = py(this).getModule("omnicompute.android")
            if (prefs.helping) m.callAttr("join", filesDir.absolutePath, prefs.helpLink, Build.MODEL.take(20), prefs.cores)
            else m.callAttr("start", filesDir.absolutePath, cloudflared(this), prefs.compute, prefs.cores, 8000,
                            !prefs.ownIp, prefs.ownUrl)
        }.start()
        return START_STICKY
    }

    private fun stopHosting() {
        Thread {
            try { py(this).getModule("omnicompute.android").callAttr("stop") } catch (_: Exception) {}
        }.apply { start(); join(12000) }
        wake?.release(); wake = null
        wifi?.release(); wifi = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        wake?.release()
        wifi?.release()
        super.onDestroy()
    }

    private fun notification(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.channel), NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1, Intent(this, HostService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notify)
            .setContentTitle(getString(if (Prefs(this).helping) R.string.helping else R.string.hosting))
            .setContentText(getString(if (Prefs(this).helping) R.string.helping_text else R.string.hosting_text))
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, getString(R.string.stop), stop).build())
            .build()
    }

    companion object {
        const val CHANNEL = "hosting"
        const val ACTION_STOP = "io.github.tosmain.omnicompute.STOP"

        @Synchronized
        fun py(context: Context): Python {
            if (!Python.isStarted()) Python.start(AndroidPlatform(context.applicationContext))
            return Python.getInstance()
        }

        /** Cloudflare's tunnel tool, built for Android and packed as a native library. */
        fun cloudflared(context: Context): String {
            val f = File(context.applicationInfo.nativeLibraryDir, "libcloudflared.so")
            return if (f.exists()) f.absolutePath else ""
        }

        fun start(context: Context) {
            context.startForegroundService(Intent(context, HostService::class.java))
        }

        fun stop(context: Context) {
            context.startService(Intent(context, HostService::class.java).setAction(ACTION_STOP))
        }
    }
}
