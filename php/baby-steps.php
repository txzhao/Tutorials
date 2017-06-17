<?php
	/* Write a program that accepts one or more numbers as command-line arguments 
	and prints the sum of those numbers to the console (stdout).*/

	// display structure information
	//var_dump($argv);
	
	$sum = 0;
	$iter = 0;
	foreach ($argv as &$value) {
		if ($iter != 0) {
			$sum += $value;
		}
		$iter ++;
	}
	echo $sum;
?>
