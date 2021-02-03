---------------------------------------------
-- drop tables and sequences
---------------------------------------------
-- drop tables
DROP TABLE Employee_Time_Keeping;
DROP TABLE Application_Table;
DROP TABLE Employee_creates_training;
DROP TABLE Employee_takes_training;
DROP TABLE Employee;
DROP TABLE Training;
DROP TABLE Role_Table;

-- drop sequences
DROP SEQUENCE employee_id_seq;
DROP SEQUENCE course_id_seq;
DROP SEQUENCE application_id_seq;
DROP SEQUENCE role_id_seq;

CREATE SEQUENCE employee_id_seq
START WITH 10000000 INCREMENT BY 1;

CREATE SEQUENCE course_id_seq
START WITH 1000 INCREMENT BY 1; 

CREATE SEQUENCE application_id_seq
START WITH 10000000 INCREMENT BY 1; 

CREATE SEQUENCE role_id_seq
START WITH 1000 INCREMENT BY 1;

-- Create tables
CREATE TABLE Training (
  Course_id                  NUMBER          default course_id_seq.NEXTVAL     PRIMARY KEY,
  Course_Name                VARCHAR(50)     NOT NULL,
  Course_Description         VARCHAR(150)    NOT NULL,
  Time_Required_mins         NUMBER          NOT NULL,
  Course_Level               NUMBER          NOT NULL
);


CREATE TABLE Role_table (
  Role_id           NUMBER            default role_id_seq.NEXTVAL     PRIMARY KEY,
  Role_Title        VARCHAR(50)       NOT NULL,
  Role_Description  VARCHAR(150)      NOT NULL,
  Compensation_hr   NUMBER            NOT NULL,
  Temporary_Flag    CHAR(1)           default 'N'     NOT NULL
);


CREATE TABLE Employee (
  Employee_id       NUMBER           default employee_id_seq.NEXTVAL      PRIMARY KEY,
  Role_id           NUMBER,
  Recruiter_Flag    CHAR(1)          default 'N'    NOT NULL,
  First_Name        VARCHAR(30)      NOT NULL,
  Last_Name         VARCHAR(30)      NOT NULL,
  Phone_number      CHAR(12)         NOT NULL,
  Social_security   CHAR(10)         NOT NULL,
  Birthdate         DATE             NOT NULL,
  First_day         DATE             NOT NULL,
  Email_Address     VARCHAR(60)      NOT NULL       UNIQUE,
  Manager_id        NUMBER,
  Warehouse_id      NUMBER,
  CONSTRAINT role_id_fk FOREIGN KEY (Role_id) REFERENCES Role_table (Role_id)
);


CREATE TABLE Employee_takes_training (
  Course_id         NUMBER      default course_id_seq.NEXTVAL,
  Employee_id       NUMBER,
  Date_started      DATE        NOT NULL,
  Date_Finished     DATE        NOT NULL,
  CONSTRAINT ett_comp_key PRIMARY KEY (Course_id, Employee_id),
  CONSTRAINT course_id_fk FOREIGN KEY (Course_id) REFERENCES Training (Course_id),
  CONSTRAINT employee_id_fk FOREIGN KEY (Employee_id) REFERENCES Employee (Employee_id)
);


CREATE TABLE Employee_creates_training (
  Course_id       NUMBER,
  Employee_id     NUMBER,
  Date_Created    DATE       NOT NULL,
  CONSTRAINT ect_comp_key PRIMARY KEY (Course_id, Employee_id),
  CONSTRAINT course_id_fk_ect FOREIGN KEY (Course_id) REFERENCES Training (Course_id),
  CONSTRAINT employee_id_fk_ect FOREIGN KEY (Employee_id) REFERENCES Employee (Employee_id)
);


CREATE TABLE Application_Table (
  Application_id     NUMBER         default application_id_seq.NEXTVAL      PRIMARY KEY,
  Role_id            NUMBER,
  Recruiter_id       NUMBER,
  First_Name         VARCHAR(30)    NOT NULL,
  Last_Name          VARCHAR(30)    NOT NULL,
  Phone_number       CHAR(12)       NOT NULL,
  Social_security    CHAR(10)       NOT NULL,
  Birthdate          DATE           NOT NULL,
  First_day          DATE           NOT NULL,
  Email_Address      VARCHAR(60)    NOT NULL     UNIQUE,
  Date_Applied       DATE           NOT NULL,
  Warehouse_id       NUMBER,
  CONSTRAINT role_id_fk_app FOREIGN KEY (Role_id) REFERENCES Role_table (Role_id),
  CONSTRAINT employee_id_fk_app FOREIGN KEY (Recruiter_id) REFERENCES Employee (Employee_id)
);


CREATE TABLE Employee_Time_Keeping (
  Work_Date                  DATE,
  Employee_id                NUMBER,
  Regular_Hours              NUMBER          NOT NULL,
  Overtime_Hours             NUMBER          default 0,
  CONSTRAINT etk_comp_key PRIMARY KEY (Work_Date, Employee_id),
  CONSTRAINT employee_id_fk_etk FOREIGN KEY (Employee_id) REFERENCES Employee (Employee_id)
);

--Insert Data

INSERT INTO Employee(Employee_id,Recruiter_Flag,First_Name,Last_Name,Phone_number,Social_security,Birthdate,First_day,Email_Address,Manager_id,Warehouse_id)
VALUES (633485,'Y','Artur','Dumberrill',3873411539,769080937,TO_DATE('10-JAN-87 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('10-JAN-17 01:02:03','DD-MON-YY HH:MI:SS'),'acottesford0@taobao.com',213343,64473);

INSERT INTO Employee(Employee_id,Recruiter_Flag,First_Name,Last_Name,Phone_number,Social_security,Birthdate,First_day,Email_Address,Manager_id,Warehouse_id)
VALUES (950774,'N','Coletta','Jemmett',6405920189,858561855,TO_DATE('19-FEB-92 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('01-JAN-20 01:02:03','DD-MON-YY HH:MI:SS'),'dkarchewski1@microsoft.com',556121,45725);

SELECT * FROM Employee;

INSERT INTO Employee_Time_Keeping(Work_Date,Employee_id,Regular_Hours,Overtime_Hours)
VALUES (TO_DATE('11-MAR-19 01:02:03','DD-MON-YY HH:MI:SS'),633485,38.93,5);

INSERT INTO Employee_Time_Keeping(Work_Date,Employee_id,Regular_Hours,Overtime_Hours)
VALUES (TO_DATE('01-JAN-20 01:02:03','DD-MON-YY HH:MI:SS'),950774,38.93,5);

SELECT * FROM Employee_Time_Keeping;

INSERT INTO Training(Course_id,Course_Name,Course_Description,Time_Required_mins,Course_Level)
VALUES (7001, 'SQL Training','Course designed to run through aspects of SQL relating to Amazon', 240, 4);

INSERT INTO Training(Course_id,Course_Name,Course_Description,Time_Required_mins,Course_Level)
VALUES (5001, 'Python Training','Course designed to run through aspects of Python relating to Amazon', 360, 2);

SELECT * FROM Training;

INSERT INTO Employee_takes_training(Course_id,Employee_id,Date_started,Date_Finished)
VALUES (7001,633485,TO_DATE('10-APR-16 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('10-MAY-16 01:02:03','DD-MON-YY HH:MI:SS'));

INSERT INTO Employee_takes_training(Course_id,Employee_id,Date_started,Date_Finished)
VALUES (5001,950774,TO_DATE('03-DEC-16 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('10-JAN-17 01:02:03','DD-MON-YY HH:MI:SS'));

SELECT * FROM Employee_takes_training;

INSERT INTO Employee_creates_training(Course_id,Employee_id,Date_Created)
VALUES (5001,633485,TO_DATE('01-JAN-12 01:02:03','DD-MON-YY HH:MI:SS'));

INSERT INTO Employee_creates_training(Course_id,Employee_id,Date_Created)
VALUES (7001,950774,TO_DATE('10-FEB-14 01:02:03','DD-MON-YY HH:MI:SS'));

SELECT * FROM Employee_creates_training;

INSERT INTO Role_table(Role_id,Role_Title,Role_Description,Compensation_hr,Temporary_Flag)
VALUES (2665,'Apple Eater','Literally paid to eat apples',50,'N');

INSERT INTO Role_table(Role_id,Role_Title,Role_Description,Compensation_hr,Temporary_Flag)
VALUES (5990,'Orange Eater','Literally paid to eat oranges',40,'Y');

SELECT * FROM Role_table;

INSERT INTO Application_Table(Application_id,Role_id,Recruiter_id,First_Name,Last_Name,Phone_number,Social_security,Birthdate,First_day,Email_Address,Date_Applied,Warehouse_id)
VALUES (2812849,2665,633485,'Abbe','Spinola',5745811330,680689001,TO_DATE('13-APR-79 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('19-JAN-18 01:02:03','DD-MON-YY HH:MI:SS'),'apeppinf@wikipedia.org',TO_DATE('15-DEC-17 01:02:03','DD-MON-YY HH:MI:SS'),78073);

INSERT INTO Application_Table(Application_id,Role_id,Recruiter_id,First_Name,Last_Name,Phone_number,Social_security,Birthdate,First_day,Email_Address,Date_Applied,Warehouse_id)
VALUES (8357842,5990,950774,'Amity','Knowling',7040819337,304695415,TO_DATE('26-JAN-98 01:02:03','DD-MON-YY HH:MI:SS'),TO_DATE('14-JUN-20 01:02:03','DD-MON-YY HH:MI:SS'),'gsalasarg@yale.edu',TO_DATE('10-NOV-19 01:02:03','DD-MON-YY HH:MI:SS'),78973);

SELECT * FROM Application_Table;

COMMIT;