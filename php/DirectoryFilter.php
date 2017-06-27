<?php
	/* class file: contains directory reading and extension filtering. */
	
    	class DirectoryFilter
    	{
        	public function filter($args)
        	{
        		$filtered_f = "";
            		foreach (new DirectoryIterator($args[1]) as $file) {
				if ($file->getExtension() == $args[2]){
					$filtered_f .= $file->getFilename() . "\n";
				}
			}
			return $filtered_f;
        	}
    	}
    
