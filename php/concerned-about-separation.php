<?php
	require_once __DIR__ . '/mymodule.php';
	$myFilter = new DirectoryFilter();
	echo $myFilter->filter($argv);