show_coverage()	{	
	for directory in $1; do
		echo $directory
		for f in $(ls $directory/*.json); do
		    python show_coverage.py $f
		done;
	done;
}

show_API() {	
        for f in $(ls $directory/*.txt); do
		    python show_API.py $f
        done;
}

show_coverage "Input"
show_API "Input"
show_coverage "Mutation"
