-- Ruffner Matthew mar9632


---------------------------------------------
-- drop tables and sequences
---------------------------------------------
-- drop tables
drop table user_topic_subsc;
drop table video_topic_link;
drop table comments;
drop table topic;
drop table creditcard;
drop table video;
drop table content_creators;
drop table user_table;

-- drop sequences
DROP SEQUENCE card_id_seq;
DROP SEQUENCE cc_id_seq;
DROP SEQUENCE user_id_seq;
DROP SEQUENCE topic_id_seq;
DROP SEQUENCE video_id_seq;
DROP SEQUENCE comment_id_seq;

---------------------------------------------
-- create sequences
---------------------------------------------
-- Create card_id_seq
CREATE SEQUENCE card_id_seq
START WITH 100000 INCREMENT BY 1;

-- Create USER_id_seq
CREATE SEQUENCE user_id_seq
START WITH 10000 INCREMENT BY 1;

-- Create cc_id_seq
CREATE SEQUENCE cc_id_seq
START WITH 10000 INCREMENT BY 1;

-- Create topic_id_seq
CREATE SEQUENCE topic_id_seq
START WITH 1000 INCREMENT BY 1;

-- Create video_id_seq
CREATE SEQUENCE video_id_seq
START WITH 100000 INCREMENT BY 1;

-- Create comment_id_seq
CREATE SEQUENCE comment_id_seq
START WITH 1 INCREMENT BY 1;

---------------------------------------------
-- create tables
---------------------------------------------
CREATE TABLE user_table
(
    user_id         NUMBER    default user_id_seq.NEXTVAL    PRIMARY KEY,
    first_name      VARCHAR(50)     NOT NULL,
    middle_name     VARCHAR(50),
    last_name       VARCHAR(50)     NOT NULL,
    email           VARCHAR(50)     UNIQUE      NOT NULL,
    systemdate      DATE            DEFAULT SYSDATE,
    birthdate       DATE            NOT NULL,
    CC_flag         VARCHAR(1)      default 'N' NOT NULL,
    CONSTRAINT  age_over13_check    CHECK ((systemdate - birthdate)/365 > 13),
    CONSTRAINT  email_length_check  CHECK (LENGTH(email) >= 7)
    );
    
CREATE TABLE content_creators
(  
    cc_id           NUMBER  default cc_id_seq.NEXTVAL     PRIMARY KEY,
    user_id         NUMBER,
    cc_username     VARCHAR(20)     UNIQUE,
    street_address  VARCHAR(30)     NOT NULL,
    city            VARCHAR(30)     NOT NULL,
    state           VARCHAR(30)     NOT NULL,
    zip_code        CHAR(5)         NOT NULL,
    state_residence VARCHAR(30)     NOT NULL,
    country_res     VARCHAR(30)     NOT NULL,
    mobile          CHAR(12)        NOT NULL,
    tier_level      VARCHAR(12)     DEFAULT 'Basic free',
    CONSTRAINT user_id_fk   FOREIGN KEY (user_id) REFERENCES user_table (user_id)
    );
    
CREATE TABLE creditcard
(
    card_id             NUMBER  default card_id_seq.NEXTVAL     PRIMARY KEY,
    contentcreator_id   NUMBER      REFERENCES content_creators (cc_id),
    card_type       VARCHAR(20)     NOT NULL,
    card_num        VARCHAR(16)     NOT NULL,
    CC_id           VARCHAR(4)      NOT NULL,
    street_billing  VARCHAR(30)     NOT NULL,
    city_billing    VARCHAR(30)     NOT NULL,
    state_billing   VARCHAR(30)     NOT NULL,
    zip_code_bill   CHAR(5)         NOT NULL
    );
    
CREATE TABLE video
(
    video_id        NUMBER  default video_id_seq.NEXTVAL    PRIMARY KEY,
    title           VARCHAR(60)     NOT NULL,
    subtitle        VARCHAR(120)    NOT NULL,
    upload_date     DATE            NOT NULL,
    video_length    NUMBER          NOT NULL,
    video_size      VARCHAR(10)     NOT NULL,
    cc_id           NUMBER          REFERENCES content_creators (cc_id),
    views           NUMBER  default '0'   NOT NULL,
    likes           NUMBER  default '0'   NOT NULL,
    revenue         NUMBER  default '0'   NOT NULL
    );
    
CREATE TABLE topic
(
    topic_id        NUMBER  default topic_id_seq.NEXTVAL    PRIMARY KEY,
    topic_name      VARCHAR(20)     NOT NULL,
    topic_desc      VARCHAR(60)     NOT NULL
    );
    
CREATE TABLE user_topic_subsc
(
    user_id         NUMBER  REFERENCES user_table (user_id),
    topic_id        NUMBER  REFERENCES topic (topic_id),
    CONSTRAINT user_topic_cpk   PRIMARY KEY (user_id, topic_id)
    );
    
CREATE TABLE video_topic_link
(
    video_id        NUMBER  REFERENCES video (video_id),
    topic_id        NUMBER  REFERENCES topic (topic_id),
    CONSTRAINT video_topic_cpk   PRIMARY KEY (video_id, topic_id)
    );
    
CREATE TABLE comments
(
    video_id        NUMBER  REFERENCES video (video_id),
    user_id         NUMBER  REFERENCES user_table (user_id),
    comment_id      NUMBER  default comment_id_seq.NEXTVAL PRIMARY KEY,
    time_date       DATE    NOT NULL,
    comment_body    VARCHAR(120)    NOT NULL
    );

-- indexes
CREATE INDEX creditcard_user_id_ix
ON creditcard (contentcreator_id);

CREATE INDEX video_cc_id_ix
ON video (cc_id);

CREATE INDEX comments_video_id_ix
ON comments (video_id);

CREATE INDEX comments_user_id_ix
ON comments (user_id);


-- start


-- 1
SELECT first_name, last_name, email, birthdate
FROM user_table
ORDER BY last_name ASC;

-- 2
SELECT first_name ||' '|| last_name AS user_full_name
FROM user_table
WHERE last_name LIKE 'K%' 
OR last_name LIKE 'L%' 
OR last_name LIKE 'M%'
ORDER BY first_name DESC;

-- 3
SELECT title, subtitle, upload_date, views, likes
FROM video
WHERE upload_date BETWEEN TO_DATE('01-JAN-20') AND TO_DATE('21-SEP-20')
ORDER BY upload_date DESC;

-- 4
SELECT title, subtitle, upload_date, views, likes
FROM video
WHERE TO_DATE('01-JAN-20') <= upload_date AND upload_date  <= TO_DATE('21-SEP-20')
ORDER BY upload_date DESC;

-- 5
SELECT video_id, 
video_size AS video_size_MB,
likes AS Likes_Earned, 
video_length AS video_length_sec, 
TRUNC((video_length/60), 1) AS video_length_min
FROM video
WHERE ROWNUM < 4
ORDER BY Likes_Earned DESC;

-- 6
SELECT title, 
likes AS Likes_Earned, 
TRUNC((video_length/60), 1) AS video_length_min
FROM video
WHERE video_length/60 >= 6 
ORDER BY Likes_Earned DESC;

-- 7
SELECT cc_id,
video_id,
likes AS Popularity, 
TRUNC(likes/5000) AS Awards,
upload_date AS Post_date
FROM video
WHERE likes/5000 > 10; 

-- 8
SELECT first_name ||' '|| last_name AS user_full_name
FROM video v  
    JOIN content_creators c ON v.cc_id = c.cc_id
    JOIN user_table u ON u.user_id = c.user_id
WHERE likes/5000 > 10; 

-- 9
SELECT SYSDATE AS today_unformatted,
TO_CHAR(SYSDATE, 'MM/DD/YYYY') AS today_formatted,
1000 AS likes,
.0325 AS pay_per_like,
10 AS pay_per_video,
1000*.0325 AS pay_sum,
10 + 1000*.0325 AS video_total
FROM DUAL;

-- 10
SELECT video_id,
likes,
.0325 AS pay_per_like,
10 AS pay_per_video,
10 + likes*.0325 AS video_total,
likes*.0325 AS pay_sum,
TO_CHAR(upload_date, 'MM/DD/YYYY') AS upload_date,
revenue
FROM video
ORDER BY revenue DESC;

-- 11
SELECT u.first_name,
u.last_name,
u.birthdate,
u.CC_flag,
c.comment_body
FROM user_table u
    JOIN comments c ON c.user_id = u.user_id
ORDER BY length(comment_body) DESC;

-- 12
SELECT u.user_id,
u.first_name ||' '|| u.last_name AS user_name,
t.topic_id,
t.topic_name
FROM topic t
    JOIN user_topic_subsc us ON us.topic_id = t.topic_id
    JOIN user_table u ON u.user_id = us.user_id
WHERE t.topic_name = 'SQL';

-- 13
SELECT v.title, v.subtitle, u.first_name, u.last_name, u.CC_flag, c.comment_body
FROM video v  
    JOIN content_creators cc ON v.cc_id = cc.cc_id
    JOIN user_table u ON u.user_id = cc.user_id
    JOIN comments c ON c.video_id = v.video_id
WHERE v.video_id = 100000 
ORDER BY u.last_name, u.first_name;

-- 14
SELECT u.first_name, u.last_name, u.email, c.comment_body
FROM user_table u  
    LEFT OUTER JOIN comments c ON c.user_id = u.user_id
WHERE c.comment_body IS NULL
ORDER BY u.last_name;

-- 15
SELECT '1-Top-Tier' AS video_tier, video_id, revenue, views
FROM video
WHERE views >= 30000
UNION
SELECT '2-Mid-Tier' AS video_tier, video_id, revenue, views
FROM video
WHERE views < 30000 AND views >= 20000
UNION
SELECT '3-Low-Tier' AS video_tier, video_id, revenue, views
FROM video
WHERE views < 20000;


-- 16
SELECT cc.cc_username, SUM(v.revenue) AS revenue, SUM(TRUNC(likes/5000)) AS Awards
FROM video v JOIN content_creators cc ON v.cc_id = cc.cc_id 
GROUP BY cc_username
ORDER BY revenue DESC;
-- So the most successful content_creator is also the one with the most awards



-- 17
SELECT DISTINCT cr.card_type,
u.first_name,
u.last_name
FROM creditcard cr
    JOIN content_creators cc ON cc.cc_id = cr.contentcreator_id
    JOIN user_table u ON u.user_id = cc.user_id;

