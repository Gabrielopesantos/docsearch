{
  description = "docsearch dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python314
            uv
            ruff
            mupdf
            pkg-config
            gcc
            stdenv.cc.cc.lib
          ];

          # Prebuilt wheels (torch, numpy) load these from the system.
          # /run/opengl-driver/lib holds the NixOS NVIDIA driver (libcuda.so) for torch.
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
        };
      }
    );
}
