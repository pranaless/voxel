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
      perSystem = { pkgs, config, ... }:
      let
        # shorthand for accessing this crate's outputs
        # you can access crate outputs under `config.nci.outputs.<crate name>` (see documentation)
        out = config.nci.outputs.voxel-render;
      in {
        nci.projects.voxel = {
          relPath = "";
        };
        nci.crates.voxel-render = {
          export = true;
        };
        devShells.default = out.devShell;
        packages.default = out.packages.release;
      };
    };
}
