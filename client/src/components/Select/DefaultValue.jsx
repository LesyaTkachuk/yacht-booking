import Typography from "@mui/material/Typography";

const DefaultValue = ({ value, lineHeight = "24px" }) => {
  return (
    <Typography variant="body1" color="secondary" lineHeight={lineHeight}>
      {value}
    </Typography>
  );
};

export default DefaultValue;
