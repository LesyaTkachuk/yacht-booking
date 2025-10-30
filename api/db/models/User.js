import { DataTypes, Sequelize } from "sequelize";
import sequelize from "../Sequelize.js";
import {
  EMAIL_REGEX,
  USER_ROLES,
  SAILING_EXPERIENCE,
} from "../../constants/auth.js";

const User = sequelize.define(
  "user",
  {
    id: {
      type: DataTypes.UUID,
      defaultValue: Sequelize.UUIDV4,
      primaryKey: true,
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true,
      validate: {
        is: EMAIL_REGEX,
      },
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    country: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    role: {
      type: DataTypes.ENUM,
      values: [USER_ROLES.LESSER, USER_ROLES.LESSEE, USER_ROLES.ADMIN],
      defaultValue: USER_ROLES.LESSEE,
    },
    sailingExp: {
      type: DataTypes.ENUM,
      values: [
        SAILING_EXPERIENCE.NONE,
        SAILING_EXPERIENCE.BEGINNER,
        SAILING_EXPERIENCE.INTERMEDIATE,
        SAILING_EXPERIENCE.PRO,
      ],
      allowNull: true,
    },
    budgetMin: {
      type: DataTypes.INTEGER,
      allowNull: true,
    },
    budgetMax: {
      type: DataTypes.INTEGER,
      allowNull: true,
    },
    hasSkipperLicense: {
      type: DataTypes.BOOLEAN,
      defaultValue: false,
    },
    avatarUrl: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    verified: {
      type: DataTypes.BOOLEAN,
      defaultValue: false,
    },
    verificationToken: {
      type: DataTypes.STRING,
      allowNull: true,
    },
    token: {
      type: DataTypes.TEXT,
      defaultValue: null,
    },
    createdAt: {
      type: DataTypes.DATE,
      defaultValue: Sequelize.NOW,
    },
    recommendations: {
      type: DataTypes.ARRAY(DataTypes.UUID),
      allowNull: true,
    },
  },
  {
    tableName: "users",
  }
);

User.sync({ alter: true });

export default User;
