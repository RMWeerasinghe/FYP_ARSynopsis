import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
// import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from "@mui/material/ListItemText";
// import InboxIcon from '@mui/icons-material/MoveToInbox';
// import MailIcon from '@mui/icons-material/Mail';

export default function SideDrawer() {
  const [open, setOpen] = React.useState(false);

  const toggleDrawer = (newOpen) => () => {
    setOpen(newOpen);
  };

  const DrawerList = (
    <Box sx={{ width: 250 }} role="presentation" onClick={toggleDrawer(false)}>
      <h4
        style={{
          fontSize: "1.9rem",
          fontWeight: 500,
          marginLeft: 18,
          marginTop: 16,
          marginBottom: 15,
        }}
      >
        History
      </h4>
      <List>
        {[
          "1074_1717151061859.pdf",
          "1544_1717586457819.pdf",
          "uml-annual-report-2020-2021.pdf",
        ].map((text, index) => (
          <ListItem key={text} disablePadding>
            <ListItemButton>
              <ListItemText
                primary={text}
                primaryTypographyProps={{
                  sx: { fontSize: "0.9rem", fontWeight: 500 },
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <div>
      <button
        style={{ backgroundColor: "transparent", borderStyle: "none" }}
        onClick={toggleDrawer(true)}
      >
        <img
          width="32"
          height="32"
          src="https://img.icons8.com/ios-filled/50/4D4D4D/menu--v6.png"
          alt="menu--v6"
        />
      </button>
      <Drawer open={open} onClose={toggleDrawer(false)}>
        {DrawerList}
      </Drawer>
    </div>
  );
}
