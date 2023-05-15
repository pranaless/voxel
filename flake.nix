{
  inputs = {
    nixpkgs.url = github:nixos/nixpkgs/nixos-unstable;
    nci = {
      url = github:yusdacra/nix-cargo-integration;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    parts = {
      url = github:hercules-ci/flake-parts;
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { parts, nci, ... }:
    parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      imports = [ nci.flakeModule ];
      perSystem = { pkgs, config, ... }: {
        nci.projects.voxel = {
          relPath = "";
        };
        nci.crates.voxel-render = {
          runtimeLibs = with pkgs; [ vulkan-loader xorg.libX11 libxkbcommon wayland ];
        };
        packages.default = config.nci.outputs.voxel-render.packages.release;
        devShells.default = config.nci.outputs.voxel.devShell.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [])
            ++ (with pkgs; [ rust-analyzer ]);
        });
      };
    };
}
