(TeX-add-style-hook
 "mystyle"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("todonotes" "colorinlistoftodos" "prependcaption" "textsize=tiny")))
   (TeX-run-style-hooks
    "amsmath"
    "amssymb"
    "mathtools"
    "amsthm"
    "graphicx"
    "subcaption"
    "array"
    "tabularx"
    "indentfirst"
    "xspace"
    "xcolor"
    "float"
    "soul"
    "booktabs"
    "makecell"
    "multirow"
    "xargs"
    "todonotes")
   (TeX-add-symbols
    '("itz" 1)
    '("itAA" 1)
    '("itA" 1)
    '("iter" 1)
    '("rest" 2)
    '("qqobjqq" 1)
    '("qobjq" 1)
    '("enscond" 2)
    '("ens" 1)
    '("choice" 1)
    '("bpa" 1)
    '("pa" 1)
    '("usup" 1)
    '("umax" 1)
    '("umin" 1)
    '("uargmax" 1)
    '("uargmin" 1)
    '("eql" 1)
    '("eq" 1)
    '("transp" 1)
    '("func" 4)
    '("absb" 1)
    '("abs" 1)
    '("normz" 1)
    '("normd" 1)
    '("normi" 1)
    '("normu" 1)
    '("normT" 1)
    '("normTV" 1)
    '("normB" 1)
    '("normb" 1)
    '("norm" 1)
    '("seg" 2)
    '("crossp" 2)
    '("dotps" 2)
    '("dotp" 2)
    '("ordin" 2)
    '("legsymb" 2)
    '("interior" 1)
    '("pdd" 2)
    '("pd" 2)
    '("wt" 1)
    '("whwh" 1)
    '("wh" 1)
    '("ol" 1)
    '("Calt" 1)
    '("ins" 1)
    '("guill" 1)
    "NN"
    "CC"
    "GG"
    "LL"
    "PP"
    "QQ"
    "RR"
    "VV"
    "ZZ"
    "FF"
    "KK"
    "TT"
    "UU"
    "EE"
    "Aa"
    "Bb"
    "Cc"
    "Dd"
    "Ee"
    "Ff"
    "Gg"
    "Hh"
    "Ii"
    "Jj"
    "Kk"
    "Ll"
    "Mm"
    "Nn"
    "Oo"
    "Pp"
    "Qq"
    "Rr"
    "Ss"
    "Tt"
    "Uu"
    "Vv"
    "Ww"
    "Xx"
    "Yy"
    "Zz"
    "al"
    "la"
    "ga"
    "Ga"
    "La"
    "si"
    "Si"
    "be"
    "de"
    "De"
    "om"
    "Om"
    "hf"
    "wtf"
    "tx"
    "tb"
    "ty"
    "tu"
    "tv"
    "tga"
    "tf"
    "dom"
    "ri"
    "realp"
    "imagp"
    "Ker"
    "Hom"
    "End"
    "tr"
    "Tr"
    "Supp"
    "Sign"
    "Corr"
    "sign"
    "cas"
    "sinc"
    "cotan"
    "Card"
    "PGCD"
    "Span"
    "Vect"
    "interop"
    "Calpha"
    "Cbeta"
    "Cal"
    "Cdeux"
    "lun"
    "ldeux"
    "linf"
    "ldeuxj"
    "Lun"
    "Ldeux"
    "Linf"
    "lzero"
    "lp"
    "Sun"
    "foralls"
    "diverg"
    "Prox"
    "Proj"
    "Id"
    "eqdef"
    "argmin"
    "argmax"
    "qandq"
    "qqandqq"
    "qifq"
    "qqifqq"
    "qwhereq"
    "qqwhereqq"
    "qwithq"
    "qqwithqq"
    "qforq"
    "qqforqq"
    "qqsinceqq"
    "qsinceq"
    "qarrq"
    "qqarrqq"
    "qiffq"
    "qqiffqq"
    "qsubjq"
    "qqsubjqq"
    "qetq"
    "qqetqq"
    "qouq"
    "qqouqq"
    "qqpourqq"
    "qavecq"
    "qqavecqq"
    "defi"
    "defeq"
    "KL"
    "oKL"
    "oE"
    "qstq"
    "ProxKL"
    "BregDiv"
    "Im"
    "ones")
   (LaTeX-add-environments
    '("rn" 1)
    "thm"
    "prop"
    "defn"
    "cor"
    "lem"
    "rem"
    "exmp"
    "rs"
    "rsfig"
    "rt")
   (LaTeX-add-lengths
    "restsubwidth"
    "restsubheight"
    "restsubmoreheight"))
 :latex)

