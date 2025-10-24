import { DataTypes, Sequelize } from "sequelize";
import sequelize from "../Sequelize.js";
import { EVENTS } from "../../constants/events.js";

const Event = sequelize.define(
  "event",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: Sequelize.UUIDV4,
      primaryKey: true,
    },
    type: {
      type: DataTypes.ENUM,
      values: [
        EVENTS.VIEW,
        EVENTS.WISHLIST,
        EVENTS.CHAT_OWNER,
        EVENTS.START_BOOKING,
        EVENTS.BOOK,
      ],
      allowNull: false,
    },
    weight: {
      type: DataTypes.INTEGER,
      defaultValue: 0,
    },
    userId: {
      type: DataTypes.UUID,
      allowNull: false,
    },
    yachtId: {
      type: DataTypes.UUID,
      allowNull: false,
    },
  },
  {
    tableName: "events",
  }
);

Event.sync({ alter: true });

export default Event;
