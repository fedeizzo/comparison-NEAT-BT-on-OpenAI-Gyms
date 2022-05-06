{ pkgs ? import <nixpkgs> { } }:

let
  mach-nix = import
    (builtins.fetchGit {
      url = "https://github.com/DavHau/mach-nix/";
      # ref = "refs/tags/3.3.0";
      ref = "master";
    })
    {
      python = "python39";
      # "https://github.com/DavHau/pypi-deps-db"
      pypiDataRev = "43f5b07a0b1614ee80723b2ad2f8e29a7b246353";
      pypiDataSha256 = "sha256:0psv5w679bgc90ga13m3pz518sw4f35by61gv7c64yl409p70rf9";
    };
  requirements =
    (builtins.replaceStrings [ "git+https://github.com/fedeizzo/neat-python.git" ] [ "" ] (builtins.readFile ./requirements.txt));
  neat = pkgs.python39Packages.buildPythonPackage {
    pname = "neat-python";
    version = "0.93";
    src = pkgs.fetchFromGitHub {
      owner = "fedeizzo";
      repo = "neat-python";
      rev = "bbc67344f90da89ef24d3866b0afc4968e653f27";
      sha256 = "sha256-qxChFC+Tr7Y6MPTuOycXLvLC5PO+Xp658ki4VMAMW+Q=";
    };
  };
  python-mach = mach-nix.mkPython {
    requirements = requirements;
    providers = {
      _defualt = "sdist";
    };
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    chromium
    python39
    python39Packages.ipython
    python39Packages.matplotlib-inline
    python39Packages.jupyter
    python39Packages.jupyter_console
    python39Packages.jupyter-client
    python39Packages.pyppeteer
  ] ++ [ python-mach neat ];
  CHROMIUM_EXECUTABLE_DERK = "$HOME/.nix-profile/bin/google-chrome-stable";
  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}";
  shellHook = ''
    export LD_PRELOAD="/run/opengl-driver/lib/libcuda.so"
  '';

}
