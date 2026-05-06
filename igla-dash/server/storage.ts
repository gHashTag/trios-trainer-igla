import { drizzle } from "drizzle-orm/better-sqlite3";
import Database from "better-sqlite3";

// SQLite path: defaults to ./data.db (works locally and on Railway with a
// mounted volume).  Override with DB_PATH for custom locations.
const dbPath = process.env.DB_PATH ?? "data.db";
const sqlite = new Database(dbPath);
sqlite.pragma("journal_mode = WAL");

export const db = drizzle(sqlite);

// IStorage placeholder kept for future expansion; the IGLA dashboard performs
// queries directly via Drizzle in routes.ts.
export interface IStorage {}
export const storage: IStorage = {};

