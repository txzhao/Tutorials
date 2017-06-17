<?php
	/* display structure information */
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