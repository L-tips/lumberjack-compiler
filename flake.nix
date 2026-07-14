{
  description = "Model compiler for the Lumberjack tree ensemble accelerator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem = { pkgs, self', ... }: {
        packages = {
          lumberjack-compiler = pkgs.rustPlatform.buildRustPackage {
            pname = "lumberjack-compiler";
            version = "0.2.1";

            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            cargoBuildFlags = [
              "--package"
              "lumberjack-compiler"
            ];
          };

          default = self'.packages.lumberjack-compiler;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
          ];
        };
      };
    };
}