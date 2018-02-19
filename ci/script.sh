set -euxo pipefail

main() {
    cargo check
    cargo test
}

main
