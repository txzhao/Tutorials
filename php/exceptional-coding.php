<?php
/* Write a program that takes an array of filepaths as arguments and outputs the basename of each, seperated by a new line.
Every file should exist but under exceptional circumstances some files may not. If this occurs, output a message similar to the below.
Unable to open file at path '/file/path' */

array_shift($argv);
foreach ($argv as $f_path) {
	try {
		$f_object = new SplFileObject($f_path);
		echo $f_object->getBasename() . "\n";
	} catch (RuntimeException $e) {
		echo "Unable to open file at path '" . $f_path . "'\n";
	}
}
