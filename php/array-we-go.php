<?php
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