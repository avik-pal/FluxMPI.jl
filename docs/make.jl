using Documenter, DocumenterMarkdown, FluxMPI

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(
  deployconfig; type="pending", repo="github.com/avik-pal/FluxMPI.jl.git")

makedocs(;
  sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true, modules=[FluxMPI],
  strict=[:doctest, :linkcheck, :parse_error, :example_block], checkdocs=:all,
  format=DocumenterMarkdown.Markdown(), draft=false, build=joinpath(@__DIR__, "docs"))

deploydocs(; repo="github.com/avik-pal/FluxMPI.jl.git",
  push_preview=true,
  deps=Documenter.Deps.pip("mkdocs", "pygments", "python-markdown-math", "mkdocs-material",
    "pymdown-extensions", "mkdocstrings", "mknotebooks",
    "pytkdocs_tweaks", "mkdocs_include_exclude_files", "jinja2"),
  make=() -> run(`mkdocs build`),
  target="site",
  devbranch="main")
