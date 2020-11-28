#!/bin/bash

echo "    Connecting to mariadb container from ubuntu container"

# mysql -h mariadb -u root -pmichelle123

if ! mysql -h mariadb -u root -pmichelle123 -e 'USE baseballdb'; then
    echo "    Could not find baseballdb. Creating data base"
    mysqladmin -u root -h mariadb -pmichelle123 create baseballdb
    echo "    Importing database from ubuntu container"
    mysql -u root -pmichelle123 -h mariadb --database=baseballdb < baseball.sql
fi

echo "    Success!"
echo "    Calculating game summaries"
# initial calculation of all players hits and at bats for each game, along with the game date
mysql -h mariadb -u root -pmichelle123 -e '
 USE baseballdb;
 CREATE TABLE IF NOT EXISTS rolling_avg_temp
	SELECT bc.game_id, bc.batter, bc.Hit, bc.atbat, gt.local_date
	FROM batter_counts bc
	INNER JOIN game_temp gt on bc.game_id = gt.game_id
	ORDER BY game_id;'

echo "    Calculating players in target game"
# finding target players for selected game and the game date
mysql -h mariadb -u root -pmichelle123 -e '
 USE baseballdb;
 CREATE TABLE IF NOT EXISTS target_players
	SELECT  bc.batter, gt.local_date
	FROM batter_counts bc
	INNER JOIN game_temp gt on bc.game_id = gt.game_id
        WHERE bc.game_id = 12560
	ORDER BY bc.batter;'

echo "    Filtering for only target players"
# using target players table to filter initial calculation for only interested players and dates before target game
mysql -h mariadb -u root -pmichelle123 -e '
 USE baseballdb; 
 CREATE TABLE IF NOT EXISTS filtered_avg
       SELECT * FROM rolling_avg_temp where batter in (select tp.batter from target_players tp);'

echo "    Calculating rolling 100 day batting avreage"
# calculating 100 day rolling average
mysql -h mariadb -u root -pmichelle123 -e '
 USE baseballdb;

 CREATE TABLE IF NOT EXISTS filtered2
 SELECT * FROM filtered_avg WHERE local_date BETWEEN "2010-12-25 00:00:00" AND "2011-04-04 00:00:00";

 CREATE TABLE IF NOT EXISTS rolling_avg
 SELECT batter, sum(Hit)/sum(atbat) as batting_avg FROM filtered2 GROUP BY batter ORDER BY batting_avg DESC, SUM(Hit)/SUM(atbat);'


echo "    Success!!!!"

# writing output
mysql -h mariadb -u root -pmichelle123 -e '
  USE baseballdb;
  SELECT * INTO OUTFILE "./outputfile.txt"
  FIELDS TERMINATED BY "," OPTIONALLY ENCLOSED BY """"
  LINES TERMINATED BY "\n"
  FROM rolling_avg;'


# cat ./output/outputfile.txt
