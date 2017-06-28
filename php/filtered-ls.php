<?php
/* Create a program that prints a list of files in a given directory, filtered by the extension of the files. 
You will be provided a directory name as the first argument to your program (e.g. '/path/to/dir/') 
and a file extension to filter by as the second argument. */

foreach (new DirectoryIterator($argv[1]) as $file) {		
	if ($file->getExtension() == $argv[2]){
		echo $file->getFilename() . "\n";
	}
}

?>
