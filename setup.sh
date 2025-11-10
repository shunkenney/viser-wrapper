# Clone git repositories that you will edit source.
git clone <url/of/lib3/git_repo> <my_proj>/submodules/<lib3>
rm -rf <my_proj>/submodules/<lib3>/.git

# Setup virtual env
uv sync
