
-- Author: Michelle An
-- RedID: 824681969

USE baseball;
	
	
-- ################# HISTORIC BATTING AVERAGE #################
DROP TABLE IF EXISTS historic_batting_avg;
	
CREATE TABLE historic_batting_avg
	SELECT batter, SUM(Hit) as total_hits, SUM(atBat) as total_at_bats
	FROM baseball.batter_counts bc
	GROUP BY batter;
	
ALTER TABLE historic_batting_avg ADD average DECIMAL (4, 3);
	
UPDATE historic_batting_avg SET average = total_hits/total_at_bats WHERE total_at_bats > 0;
UPDATE historic_batting_avg SET average = 0.000 WHERE total_at_bats = 0;
	
SELECT * FROM historic_batting_avg;
	

-- ################# ANNUAL BATTING AVERAGE #################
SET @year_end_date = DATE('2008-12-31'); -- use this variable to set the last day to include in the annual search window
	
DROP TABLE IF EXISTS annual_avg_temp;
DROP TABLE IF EXISTS annual_batting_avg;
	
CREATE TABLE annual_avg_temp
	SELECT bc.batter, bc.Hit, bc.atBat, gt.local_date
	FROM batter_counts bc
	INNER JOIN game_temp gt ON bc.game_id = gt.game_id
	WHERE gt.local_date BETWEEN DATE_ADD(@year_end_date, INTERVAL -1 YEAR) AND @year_end_date;
	
SELECT * FROM annual_avg_temp ORDER BY batter;
	
CREATE TABLE annual_batting_avg
	SELECT batter, SUM(Hit) as total_hits, SUM(atBat) as total_at_bats
	FROM annual_avg_temp
	GROUP BY batter;
	
ALTER TABLE annual_batting_avg ADD annual_batting_average decimal (4,3);
	
UPDATE annual_batting_avg SET annual_batting_average = FORMAT((total_hits/total_at_bats), 3) WHERE total_at_bats > 0;
UPDATE annual_batting_avg SET annual_batting_average = 0 WHERE total_at_bats = 0;
	
SELECT * FROM annual_batting_avg;
	
	
-- ################# ROLLING BATTING AVERAGE #################
SET @date_range = 100; -- use this variable to set the range to look through for the rolling averages

DROP TABLE IF EXISTS rolling_avg_temp;
DROP TABLE IF EXISTS rolling_batting_avg;
	
CREATE TABLE rolling_avg_temp
	SELECT bc.game_id, bc.batter, bc.Hit, bc.atbat, gt.local_date
	FROM batter_counts bc
	INNER JOIN game_temp gt on bc.game_id = gt.game_id
	ORDER BY game_id;
	
SELECT * FROM rolling_avg_temp rat ORDER BY game_id DESC;
	
	
CREATE TABLE rolling_batting_avg
	SELECT rat1.game_id, rat1.local_date AS game_date, rat1.batter,
	IFNULL((SELECT SUM(Hit) FROM rolling_avg_temp rat2 WHERE rat2.local_date > DATE_ADD(rat1.local_date, INTERVAL - @date_range DAY) AND rat2.local_date < rat1.local_date AND rat1.batter = rat2.batter), 0) AS last_100_days_hits,
	IFNULL((SELECT SUM(atbat) FROM rolling_avg_temp rat2 WHERE rat2.local_date > DATE_ADD(rat1.local_date, INTERVAL - @date_range DAY) AND rat2.local_date < rat1.local_date AND rat1.batter = rat2.batter), 0) AS last_100_days_atbats
	FROM rolling_avg_temp rat1
	ORDER BY rat1.game_id DESC
	LIMIT 100
	;
	
ALTER TABLE rolling_batting_avg ADD rolling_average decimal (4,3);
	
UPDATE rolling_batting_avg SET rolling_average = FORMAT((last_100_days_hits /last_100_days_atbats), 3) WHERE last_100_days_atbats > 0;
UPDATE rolling_batting_avg SET rolling_average = 0 WHERE last_100_days_atbats = 0;
	
SELECT * FROM rolling_batting_avg rba;

