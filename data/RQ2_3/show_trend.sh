show_coverage()	{
	namelist=("FreeFuzz" "FreeFuzz-DBMu" "FreeFuzz-RandMu" "FreeFuzz-TypeMu")
	for directory in ${namelist[@]}; do
		echo $1-$directory
		python show_coverage.py $1/$directory
	done;
}

show_coverage pytorch
show_coverage tensorflow
