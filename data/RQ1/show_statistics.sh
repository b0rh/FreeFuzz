show_coverage()	{	
	for directory in $1; do
		echo $directory
        for f in $(ls $directory/*.json); do
		    python show_statistics.py $f
        done;
	done;
}

show_coverage "pytorch"
show_coverage "tensorflow"
