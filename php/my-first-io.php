<?php
	/* Write a program that uses a single filesystem operation to read a file 
	and print the number of newlines (\n) it contains to the console (stdout). */

	$f_contents = file_get_contents($argv[1]);
	echo substr_count($f_contents, "\n");

?>