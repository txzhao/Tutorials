<?php
/* Write a program that takes an array of filepaths as arguments, filtering out files that do not exist 
and mapping existing files to SplFileObject's. Finally output the basename of the files, each on a new line. */

function createSplFileObject($f_name)
{
	return new SplFileInfo($f_name);;
}

$f_path = array_shift($argv);
$f_exist = array_filter($argv, "file_exists");
$f_object = array_map("createSplFileObject", $f_exist);
foreach ($f_object as $f_ob) {
	echo $f_ob->getBasename() . "\n";
}
