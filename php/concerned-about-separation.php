<?php
	/*Create a program that prints a list of files in a given directory, filtered by the extension of the files. 
	The first argument is the directory name and the second argument is the extension filter. 
	Print the list of files (one file per line) to the console.*/

	require_once __DIR__ . '/DirectoryFilter.php';
	$myFilter = new DirectoryFilter();
	echo $myFilter->filter($argv);
