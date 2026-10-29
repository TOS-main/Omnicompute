package io.github.tosmain.omnicompute

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.view.Menu
import android.view.MenuItem
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import java.io.File

/** The hub's own web page (chat, models, devices, links), served by the service on this phone. */
class MainActivity : Activity() {
    private lateinit var web: WebView
    private val ui = Handler(Looper.getMainLooper())
    private var upload: ValueCallback<Array<Uri>>? = null
    private var loaded = false

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        HostService.start(this)  // hosting is what the app is for: opening it starts the mesh

        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.settings.mediaPlaybackRequiresUserGesture = false
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val host = request.url.host ?: return false
                if (host == "127.0.0.1" || host == "localhost") return false
                startActivity(Intent(Intent.ACTION_VIEW, request.url))  // GitHub, Hugging Face, ... in the browser
                return true
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (request.isForMainFrame) { loaded = false; waiting(); ui.postDelayed({ open() }, 1500) }
            }
        }
        web.webChromeClient = object : WebChromeClient() {
            // "Import a GGUF file" on the Models tab
            override fun onShowFileChooser(view: WebView, callback: ValueCallback<Array<Uri>>, params: FileChooserParams): Boolean {
                upload?.onReceiveValue(null)
                upload = callback
                return try { startActivityForResult(params.createIntent(), 2); true } catch (_: Exception) { upload = null; false }
            }
        }
        setContentView(web)
        waiting()
        ui.postDelayed({ open() }, 1500)
    }

    private fun home() = filesDir.absolutePath
    private fun key(): String = File(filesDir, ".omnicompute/mesh.key").takeIf { it.exists() }?.readText()?.trim() ?: ""

    private fun waiting() {
        web.loadDataWithBaseURL(null, "<body style='font:16px sans-serif;background:#111318;color:#ddd;display:grid;place-items:center;height:90vh'>" +
            "<p>${getString(R.string.starting)}</p></body>", "text/html", "utf-8", null)
    }

    private fun open() {
        val k = key()
        if (k.isEmpty()) { ui.postDelayed({ open() }, 1000); return }  // the first start makes the key
        if (!loaded) { loaded = true; web.loadUrl("http://127.0.0.1:8000/?k=" + Uri.encode(k)) }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == 2) {
            upload?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(resultCode, data))
            upload = null
        } else super.onActivityResult(requestCode, resultCode, data)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (web.canGoBack()) web.goBack() else moveTaskToBack(true)  // keeps hosting
    }

    // ------------------------------------------------------------------ menu

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        val p = Prefs(this)
        menu.add(0, 1, 0, R.string.share).setShowAsAction(MenuItem.SHOW_AS_ACTION_IF_ROOM)
        menu.add(0, 2, 1, R.string.share_full)
        menu.add(0, 3, 2, R.string.compute).setCheckable(true).isChecked = p.compute
        menu.add(0, 4, 3, R.string.boot).setCheckable(true).isChecked = p.startAtBoot
        menu.add(0, 5, 4, R.string.battery)
        menu.add(0, 8, 5, R.string.connection)
        menu.add(0, 7, 5, R.string.use_key)
        menu.add(0, 6, 6, R.string.stop_hosting)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val p = Prefs(this)
        when (item.itemId) {
            1 -> share(pyCall("permanent_link"))
            2 -> share(fullLink())
            3 -> { p.compute = !p.compute; item.isChecked = p.compute; restart() }
            4 -> { p.startAtBoot = !p.startAtBoot; item.isChecked = p.startAtBoot }
            5 -> askBattery()
            6 -> { HostService.stop(this); finishAndRemoveTask() }
            7 -> askKey()
            8 -> askConnection()
            else -> return super.onOptionsItemSelected(item)
        }
        return true
    }

    private fun pyCall(name: String): String =
        try { HostService.py(this).getModule("omnicompute.android").callAttr(name, home()).toString() } catch (_: Exception) { "" }

    private fun fullLink(): String {
        val k = key()
        val pages = pyCall("pages")
        return if (k.isEmpty() || pages.isEmpty()) "" else "$pages/go.html#k=$k"
    }

    private fun share(text: String) {
        if (text.isEmpty()) return
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text), null))
    }

    /** Move an existing mesh here: same key, so every link and device keeps working. */
    private fun askKey() {
        val input = android.widget.EditText(this).apply { hint = "mesh key or full link"; setSingleLine() }
        android.app.AlertDialog.Builder(this)
            .setTitle(R.string.use_key)
            .setMessage(R.string.use_key_text)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val why = try {
                    HostService.py(this).getModule("omnicompute.android").callAttr("set_key", home(), input.text.toString()).toString()
                } catch (e: Exception) { e.toString() }
                if (why.isEmpty()) restart() else android.widget.Toast.makeText(this, why, android.widget.Toast.LENGTH_LONG).show()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    /** Cloudflare (any network, the default) or this phone's own address. */
    private fun askConnection() {
        val p = Prefs(this)
        val choices = arrayOf(getString(R.string.conn_cloud), getString(R.string.conn_own))
        android.app.AlertDialog.Builder(this)
            .setTitle(R.string.connection)
            .setSingleChoiceItems(choices, if (p.ownIp) 1 else 0) { d, which ->
                d.dismiss()
                if (which == 0) { p.ownIp = false; restart(); return@setSingleChoiceItems }
                val input = android.widget.EditText(this).apply { hint = "http://your-public-ip:8000 (optional)"; setSingleLine(); setText(p.ownUrl) }
                android.app.AlertDialog.Builder(this)
                    .setTitle(R.string.conn_own)
                    .setMessage(R.string.conn_own_text)
                    .setView(input)
                    .setPositiveButton(android.R.string.ok) { _, _ -> p.ownIp = true; p.ownUrl = input.text.toString().trim(); restart() }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
            }
            .show()
    }

    private fun restart() {
        HostService.stop(this)
        ui.postDelayed({ HostService.start(this); loaded = false; open() }, 3000)
    }

    @SuppressLint("BatteryLife")
    private fun askBattery() {
        val pm = getSystemService(PowerManager::class.java)
        if (pm.isIgnoringBatteryOptimizations(packageName)) {
            startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
        } else {
            startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
        }
    }
}
