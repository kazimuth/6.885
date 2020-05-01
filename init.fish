echo '~~ setting up adv-lth fish... ~~'
set DIR (realpath (dirname (status --current-file)))

if [ (hostname) = "phenocryst" ]
  . ~/.local/miniconda3/etc/fish/conf.d/conda.fish
end

conda activate $DIR/miniconda
