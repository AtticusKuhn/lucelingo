{ pkgs ? import <nixpkgs-unstable> {} }:
let
  # Pin to a single Python toolchain to avoid mixing 3.12/3.13 site-packages
  py = pkgs.python312;
  pyPkgs = pkgs.python312Packages;
  myPython = py.withPackages (ps: with pyPkgs; [
    # Core deps
    openai
    pydantic
    pydantic-core
    langchain
    langchain-community

    # Dev tools (keep on same interpreter to avoid PYTHONPATH contamination)
    black
    ruff
  ]);
in
pkgs.mkShell {
  packages = [
    myPython
    pkgs.unzip
    pkgs.git
    pkgs.sqlitebrowser
    pkgs.sqlite
    pkgs.codex
  ];
}
