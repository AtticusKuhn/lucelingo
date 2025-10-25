{ pkgs ? import <nixpkgs-unstable> {} }:
with pkgs;
let
my-python-packages = ps: with ps; [
  ];
  my-python = pkgs.python3.withPackages my-python-packages;
in
mkShell {
  buildInputs = [
    my-python
    pkgs.unzip
    pkgs.aider-chat
    pkgs.codex
    black                           # Code formatter
    ruff                            # Linter
    git
    # Add other development tools as needed
    sqlitebrowser 
    sqlite
  ];
}
