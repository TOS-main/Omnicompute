package io.github.tosmain.edgemesh

import android.content.Context

/** The app's few settings. Hosting is the point of the app; computing is an opt-in side job. */
class Prefs(context: Context) {
    private val p = context.getSharedPreferences("edgemesh", Context.MODE_PRIVATE)

    var compute: Boolean
        get() = p.getBoolean("compute", false)
        set(v) = p.edit().putBoolean("compute", v).apply()

    var cores: Int  // 0 = all; the web page's core picker sets it for a running host
        get() = p.getInt("cores", 0)
        set(v) = p.edit().putInt("cores", v).apply()

    var startAtBoot: Boolean
        get() = p.getBoolean("boot", true)
        set(v) = p.edit().putBoolean("boot", v).apply()

    var hosting: Boolean  // false after the user pressed Stop, so a reboot doesn't start it again
        get() = p.getBoolean("hosting", true)
        set(v) = p.edit().putBoolean("hosting", v).apply()
}
