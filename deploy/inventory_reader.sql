-- Role used only for inventory text-to-SQL. Run after the units table exists.
-- Set its password separately: ALTER ROLE inventory_reader PASSWORD '...';
CREATE ROLE inventory_reader LOGIN;
GRANT USAGE ON SCHEMA public TO inventory_reader;
GRANT SELECT ON units TO inventory_reader;
