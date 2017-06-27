<?php
	require_once __DIR__ . '/DirectoryFilter.php';
	$myFilter = new DirectoryFilter();
	echo $myFilter->filter($argv);
