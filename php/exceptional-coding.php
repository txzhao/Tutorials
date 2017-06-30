<?php

array_shift($argv);
foreach ($argv as $f_path) {
	try {
		$f_object = new SplFileObject($f_path);
		echo $f_object->getBasename() . "\n";
	} catch (RuntimeException $e) {
		echo "Unable to open file at path '" . $f_path . "'\n";
	}
}
