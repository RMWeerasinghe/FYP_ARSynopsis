import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import React from 'react';


function VerticallyCenteredModal(props) {
    return (
      <Modal
        {...props}
        size="lg"
        aria-labelledby="contained-modal-title-vcenter"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title id="contained-modal-title-vcenter">
            Abstractive Summary
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {props.children}
        </Modal.Body>
        <Modal.Footer>
          <Button onClick={props.onHide} style={{color: "#495057", backgroundColor: "#fff" ,border: "1px solid #ced4da"}}>Close</Button>
        </Modal.Footer>
      </Modal>
    );
  }

  export default VerticallyCenteredModal;