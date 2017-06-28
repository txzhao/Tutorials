<?php
/* Write a program that uses a single filesystem operation to read a file 
and print the number of newlines (\n) it contains to the console (stdout). */

$f_contents = file_get_contents($argv[1]);	// read a file into a string
echo substr_count($f_contents, "\n");		// count the number of substring occurances

?>
