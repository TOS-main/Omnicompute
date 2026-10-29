// The EdgeMesh app: Python (Chaquopy) runs the same hub as `edgemesh host --public`, a
// foreground service keeps it alive with the screen off, and the screen is a web view of the
// hub's own page. cloudflared (built for Android by .github/workflows/android.yml) gives it an
// address that works from any network.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

val edgemeshVersion = Regex("""version = "([^"]+)"""").find(rootProject.file("../pyproject.toml").readText())!!.groupValues[1]

android {
    namespace = "io.github.tosmain.edgemesh"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.github.tosmain.edgemesh"
        minSdk = 26
        targetSdk = 34
        versionName = edgemeshVersion
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    signingConfigs {
        create("release") {
            // A private key from the repository secrets when there is one; otherwise the shared
            // key in this folder, so every build of the open-source app installs over the last.
            val own = System.getenv("EDGEMESH_KEYSTORE")
            storeFile = if (own != null) file(own) else rootProject.file("shared-release.keystore")
            storePassword = System.getenv("EDGEMESH_KEYSTORE_PASSWORD") ?: "edgemesh"
            keyAlias = System.getenv("EDGEMESH_KEY_ALIAS") ?: "edgemesh"
            keyPassword = System.getenv("EDGEMESH_KEY_PASSWORD") ?: "edgemesh"
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

// the edgemesh package itself, from the repository root
val copyEdgemesh by tasks.registering(Copy::class) {
    from(rootProject.file("../edgemesh")) { exclude("**/__pycache__/**", "**/*.pyc") }
    into(layout.buildDirectory.dir("python-src/edgemesh"))
}
tasks.named("preBuild") { dependsOn(copyEdgemesh) }

chaquopy {
    defaultConfig {
        version = "3.11"
        extractPackages("edgemesh")  // the hub reads its web page and installers as files
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
