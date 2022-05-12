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
      pypiDataRev = "4b53a8a9b3ce7266d8d860d0ef64d969dbc94515";
      pypiDataSha256 = "sha256:1ckfh8f2gxsdnvgl58bdbks9pnzq9p1rv80fjkasag7hirkzk7gx";
    };
  requirements =
    (builtins.replaceStrings [ "neat-python @ git+https://github.com/fedeizzo/neat-python.git@bbc67344f90da89ef24d3866b0afc4968e653f27" ] [ "" ] (builtins.readFile ./requirements.txt));
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
    python39Packages.matplotlib
    python39Packages.matplotlib-inline
    python39Packages.jupyter
    python39Packages.jupyter_console
    python39Packages.jupyter-client
    python39Packages.jupyterlab
    python39Packages.jupyterlab-widgets
    python39Packages.jupyterlab-pygments
    python39Packages.pyppeteer
  ] ++ [ python-mach neat ];
  CHROMIUM_EXECUTABLE_DERK = "$HOME/.nix-profile/bin/google-chrome-stable";
  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}";
  shellHook = ''
    export LD_PRELOAD="/run/opengl-driver/lib/libcuda.so"
  '';

}
