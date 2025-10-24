import sequelize from "../Sequelize.js";
import User from "./User.js";
import Yacht from "./Yacht.js";
import Event from "./Event.js";

// Users ↔ Yachts
User.hasMany(Yacht, { foreignKey: "userId", as: "yachts" });
Yacht.belongsTo(User, { foreignKey: "userId", as: "user" });

// Users ↔ Events
User.hasMany(Event, { foreignKey: "userId", as: "events" });
Event.belongsTo(User, { foreignKey: "userId", as: "user" });

// Yachts ↔ Events
Yacht.hasMany(Event, { foreignKey: "yachtId", as: "events" });
Event.belongsTo(Yacht, { foreignKey: "yachtId", as: "yacht" });

// Sync once — after all associations
await sequelize.sync({ alter: true });

export { sequelize, User, Yacht, Event };
