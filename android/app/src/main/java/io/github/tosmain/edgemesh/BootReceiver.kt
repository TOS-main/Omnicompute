package io.github.tosmain.edgemesh

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Hosting starts again after a restart or an app update, unless the user turned that off or pressed Stop. */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val prefs = Prefs(context)
        if (prefs.startAtBoot && prefs.hosting) HostService.start(context)
    }
}
