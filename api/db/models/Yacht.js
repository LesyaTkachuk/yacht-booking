import { DataTypes, Sequelize } from "sequelize";
import sequelize from "../Sequelize.js";

const Yacht = sequelize.define(
  "yacht",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: Sequelize.UUIDV4,
      primaryKey: true,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    type: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    guests: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: 0,
    },
    cabins: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: 0,
    },
    crew: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: 0,
    },
    length: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    year: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: 0,
    },
    model: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    summerLowSeasonPrice: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    summerHighSeasonPrice: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    winterLowSeasonPrice: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    winterHighSeasonPrice: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    description: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    rating: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: true,
      defaultValue: 0,
    },
    country: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    baseMarina: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    photos: {
      type: DataTypes.ARRAY(DataTypes.STRING),
      allowNull: true,
    },
    userId: {
      type: DataTypes.UUID,
      allowNull: false,
    },
    similarYachts: {
      type: DataTypes.ARRAY(DataTypes.UUID),
      allowNull: true,
    },
  },
  {
    tableName: "yachts",
  }
);

Yacht.sync({ alter: true });

export default Yacht;
