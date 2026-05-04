plugins {
    `java-library`
}

group = "dev.stepback"
version = "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
    withSourcesJar()
    withJavadocJar()
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("-Xlint:all"))
}

tasks.test {
    useJUnitPlatform()
    // The fixture corpus lives outside the jvm binding tree, in
    // <repo-root>/stepback-core/fixtures/v1. Resolve and pin it here
    // so test code never has to guess project layout.
    val fixturesDir = rootDir.resolve("../../stepback-core/fixtures/v1").canonicalFile
    systemProperty("stepback.fixtures", fixturesDir.absolutePath)
    testLogging {
        events("passed", "failed", "skipped")
        showStandardStreams = false
    }
}
