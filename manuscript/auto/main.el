(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "latin1")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "hyperref"
    "tikz"
    "inputenc"
    "url"
    "fullpage"
    "cite"
    "caption"
    "bm"
    "mystyle"
    "amsmath"
    "graphicx")
   (TeX-add-symbols
    '("smtr" 3)
    '("bmtr" 3)
    '("ubmr" 2)
    '("ubm" 1)
    '("ubar" 1)
    "midarrow")
   (LaTeX-add-labels
    "eq:ml-of-hmm"
    "eq:em-q-funciton"
    "eq:em-m-opt"
    "eq:m-step-subs"
    "eq:init-distribution-update"
    "eq:transition-update"
    "eq:generative-model-update"
    "eq:gm-update"
    "eq:sub-gm")
   (LaTeX-add-bibliographies
    "myref"))
 :latex)

