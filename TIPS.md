# Tips

## Some hints and suggestions if the README does not work.

### Using a specific version.

If you want a specific tagged version, from the command line do:
```
$ git tag -l # To see available tags, followed by:
$ git checkout tags/<tag_name> # or simply:
$ git checkout v0.0.1
```
Restart the Julia project as shown in the [README](https://github.com/StanJulia/ROSStanPluto.jl/blob/master/README.md).

### Instantiating environments (typically only needed once).

Pre Julia-1.7, `julia --project=ROSStanPluto` is not available, use:
```Julia
julia> ]                               # Actvate Pkg mode
(@v1.7) pkg> activate .                # Activate pkg in .
(ROSStanPluto) pkg> instantiate        # Install in pkg environment
(ROSStanPluto) pkg> <delete>           # Exit package mode
julia>
```

If the above procedure fails, if present, try to delete the Manifest.toml file and repeat the above steps. As mentioned above, these steps are only needed the first time.

### Using projects after the initial install.

To start Julia "in a project" pre Julia-1.7 I have defined in my shell startup file (e.g. ~/.zshrc):
```Julia
alias j="clear; /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia"
alias ros="clear; cd ~/.julia/dev/ROSStanPluto; j -i -e 'using Pkg; ```
and then type:
```Julia
$ ros
julia> cd("notebooks"); import Pluto; Pluto.run()
```
