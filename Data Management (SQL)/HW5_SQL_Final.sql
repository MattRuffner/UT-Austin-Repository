    -- Start --

-- mar9632 Matthew Ruffner

-- Question 1
-- current user table and the prospective user table have the following columns in common: first name, last name, email

-- Question 2
DROP TABLE client_dw;
CREATE TABLE client_dw (
    data_source     VARCHAR(4),
    client_id       NUMBER,
    first_name       VARCHAR(20)         NOT NULL,
    last_name        VARCHAR(20)         NOT NULL,
    email            VARCHAR(50) UNIQUE  NOT NULL,
    status           CHAR(1) DEFAULT 'N' NOT NULL,
    CONSTRAINT PK_client_source PRIMARY KEY (data_source, client_id)
);

-- Question 3
CREATE OR REPLACE VIEW curr_user_view
AS
SELECT 'curr' AS data_source, user_id AS client_id, First_name, Last_name,
email, CC_flag AS status
FROM curr_user_table;

CREATE OR REPLACE VIEW prospective_user_view
AS
SELECT 'pros' AS data_source, prospective_id AS client_id,
PC_first_name AS First_name, PC_last_name as Last_name, email, 'N' AS status
FROM prospective_user;


-- Question 4
-- inserting new data into data warehouse:
INSERT INTO client_dw
SELECT curr.*
FROM curr_user_view curr LEFT JOIN client_dw dw
   ON curr.data_source = dw.data_source
   AND curr.client_id = dw.client_id
WHERE dw.client_id IS NULL;

INSERT INTO client_dw
SELECT pros.*
FROM prospective_user_view pros LEFT JOIN client_dw dw
   ON pros.data_source = dw.data_source
   AND pros.client_id = dw.client_id
WHERE dw.client_id IS NULL;

-- Question 5
MERGE INTO client_dw cdw
   USING prospective_user_view puv
   ON (cdw.client_id = puv.client_id AND cdw.data_source = puv.data_source)
   WHEN MATCHED THEN
       UPDATE SET
           cdw.first_name = puv.first_name,
           cdw.last_name = puv.last_name,
           cdw.email = puv.email,
           cdw.status = puv.status
   WHEN NOT MATCHED THEN
   INSERT (first_name, last_name, email, status)
   VALUES (puv.first_name, puv.last_name, puv.email, puv.status);

MERGE INTO client_dw cdw
   USING curr_user_view cuv
   ON (cdw.client_id = cuv.client_id AND cdw.data_source = cuv.data_source)
   WHEN MATCHED THEN
       UPDATE SET
           cdw.first_name = cuv.first_name,
           cdw.last_name = cuv.last_name,
           cdw.email = cuv.email,
           cdw.status = cuv.status
   WHEN NOT MATCHED THEN
   INSERT (first_name, last_name, email, status)
   VALUES (cuv.first_name, cuv.last_name, cuv.email, cuv.status);

-- Question 6
CREATE OR REPLACE PROCEDURE user_etl_proc
AS
BEGIN
-- inserting data:
    INSERT INTO client_dw
    SELECT curr.*
    FROM curr_user_view curr LEFT JOIN client_dw dw
        ON curr.data_source = dw.data_source
        AND curr.client_id = dw.client_id
    WHERE dw.client_id IS NULL;

    INSERT INTO client_dw
    SELECT pros.*
    FROM prospective_user_view pros LEFT JOIN client_dw dw
        ON pros.data_source = dw.data_source
        AND pros.client_id = dw.client_id
    WHERE dw.client_id IS NULL;

-- updating data:
    MERGE INTO client_dw cdw
    USING prospective_user_view puv
    ON (cdw.client_id = puv.client_id AND cdw.data_source = puv.data_source)
    WHEN MATCHED THEN
        UPDATE SET
            cdw.first_name = puv.first_name,
            cdw.last_name = puv.last_name,
            cdw.email = puv.email,
            cdw.status = puv.status
    WHEN NOT MATCHED THEN
    INSERT (first_name, last_name, email, status)
    VALUES (puv.first_name, puv.last_name, puv.email, puv.status);

    MERGE INTO client_dw cdw
    USING curr_user_view cuv
    ON (cdw.client_id = cuv.client_id AND cdw.data_source = cuv.data_source)
    WHEN MATCHED THEN
        UPDATE SET
            cdw.first_name = cuv.first_name,
            cdw.last_name = cuv.last_name,
            cdw.email = cuv.email,
            cdw.status = cuv.status
    WHEN NOT MATCHED THEN
    INSERT (first_name, last_name, email, status)
    VALUES (cuv.first_name, cuv.last_name, cuv.email, cuv.status);
END;
/

EXECUTE user_etl_proc;
