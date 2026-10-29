// The Omnicompute app: Python (Chaquopy) runs the same hub as `omnicompute host --public`, a
// foreground service keeps it alive with the screen off, and the screen is a web view of the
// hub's own page. cloudflared (built for Android by .github/workflows/android.yml) gives it an
// address that works from any network.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val omnicomputeVersion = Regex("""version = "([^"]+)"""").find(rootProject.file("../pyproject.toml").readText())!!.groupValues[1]

android {
    namespace = "io.github.tosmain.omnicompute"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.github.tosmain.omnicompute"
        minSdk = 26
        targetSdk = 34
        versionName = omnicomputeVersion
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    signingConfigs {
        create("release") {
            // A private key from the repository secrets when there is one; otherwise the shared
            // key in this folder, so every build of the open-source app installs over the last.
            val own = System.getenv("OMNICOMPUTE_KEYSTORE")
            storeFile = if (own != null) file(own) else rootProject.file("shared-release.keystore")
            storePassword = System.getenv("OMNICOMPUTE_KEYSTORE_PASSWORD") ?: "edgemesh"  // the shared keystore was made under the old name
            keyAlias = System.getenv("OMNICOMPUTE_KEY_ALIAS") ?: "edgemesh"  // the shared keystore was made under the old name
            keyPassword = System.getenv("OMNICOMPUTE_KEY_PASSWORD") ?: "edgemesh"  // the shared keystore was made under the old name
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // cloudflared ships as libcloudflared.so so Android unpacks it somewhere it may run from
    packaging { jniLibs { useLegacyPackaging = true } }
}

// the omnicompute package itself, from the repository root
val copyOmnicompute by tasks.registering(Copy::class) {
    from(rootProject.file("../omnicompute")) { exclude("**/__pycache__/**", "**/*.pyc") }
    into(layout.buildDirectory.dir("python-src/omnicompute"))
}
tasks.configureEach { if (name.contains("PythonSources")) dependsOn(copyOmnicompute) }

chaquopy {
    defaultConfig {
        version = "3.11"
        extractPackages("omnicompute")  // the hub reads its web page and installers as files
        pip {
            install("numpy")
            install("cryptography")
            install("aiohttp")
            install("jinja2")
            install("segno")
            install("threadpoolctl")
            install("gguf")
        }
    }
    sourceSets {
        getByName("main") {
            srcDir(layout.buildDirectory.dir("python-src").get().asFile)
        }
    }
}
