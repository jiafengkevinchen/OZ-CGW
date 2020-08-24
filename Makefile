all:
	cat settings_lic_true.json > settings.json; sh master.sh; cat settings_lic_false.json > settings.json; sh master.sh
