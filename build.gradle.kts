plugins {
    id("java")
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "dev.jacobandersen"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("ai.djl:api:0.24.0")
    implementation("ai.djl.pytorch:pytorch-engine:0.24.0")
    implementation("ai.djl.huggingface:tokenizers:0.24.0")
    implementation("org.slf4j:slf4j-simple:1.7.32")
}

tasks.withType<Jar> {
    manifest {
        attributes["Main-Class"] = "dev.jacobandersen.sentimentanalyzer.SentimentAnalyzer"
    }
}