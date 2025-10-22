import { Typography } from "@mui/material";

const DetailItem = ({ icon: Icon, keyTitle, value }) => {
  return (
    <Stack alignItems={"center"} justifyContent={"center"} width={"19%"}>
      <Stack
        direction={"row"}
        gap={1}
        alignItems={"center"}
        justifyContent={"center"}
      >
        <Icon />
        <Typography overflow={"hidden"}>{value}</Typography>
      </Stack>
      <Typography color="secondary">{keyTitle}</Typography>
    </Stack>
  );
};

export default DetailItem;
